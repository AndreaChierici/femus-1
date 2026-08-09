// W^{1,inf} over Omega (groups 7, 8), reporting the gradient max in BOTH
// fixed-distance bands and h-SCALED bands (d >= m h).
//
// The h-scaled part is the one that decides the restatement: the theory
// would exclude a layer of width O(h), so what must be flat in ml is
// sup{ |grad w_h| : dist(x, dOmega) >= m h } at fixed m, not at fixed
// physical distance.
//
// Call site, right after system.MGsolve():
//     GetW1infNorm(mlSol, "u", 0.2 * pow(0.5, numberOfUniformLevels - 1));
//
// Geometry of martaTest4*: Omega = [-xb, xb] x [-yb, yb].

void GetW1infNorm(MultiLevelSolution & mlSol, const char solName[], const double h) {

    const unsigned level = mlSol._mlMesh->GetNumberOfLevels() - 1;
    Mesh* msh = mlSol._mlMesh->GetLevel(level);
    Solution* sol = mlSol.GetSolutionLevel(level);
    const unsigned dim = msh->GetDimension();
    unsigned xType = 2;

    unsigned soluIndex = mlSol.GetIndex(solName);
    unsigned soluType = mlSol.GetSolutionType(soluIndex);
    unsigned iproc = msh->processor_id();

    const double xb = 0.6, yb = 0.4;                      // dOmega

    const unsigned nf = 5;
    const double fEdge[nf] = {0.0, 0.025, 0.05, 0.1, 0.2};       // fixed bands

    const unsigned nm = 5;
    const double mult[nm] = {0.5, 1.0, 2.0, 4.0, 8.0};           // d >= m h

    std::vector<double> fMax(nf, 0.), mMax(nm, 0.);
    std::vector<std::vector<double>> fArg(nf, std::vector<double>(dim, 0.));
    std::vector<std::vector<double>> mArg(nm, std::vector<double>(dim, 0.));
    double maxU = 0.;

    std::vector<double> phi, phi_x;
    double weight;

    for(int iel = msh->_elementOffset[iproc]; iel < msh->_elementOffset[iproc + 1]; iel++) {

        short unsigned ielGroup = msh->GetElementGroup(iel);
        if(ielGroup == 5 || ielGroup == 6 || ielGroup == 9) continue;   // Omega only

        short unsigned ielGeom = msh->GetElementType(iel);
        unsigned nDofu = msh->GetElementDofNumber(iel, soluType);
        unsigned nDofx = msh->GetElementDofNumber(iel, xType);

        std::vector<std::vector<double>> x1(dim);
        for(unsigned k = 0; k < dim; k++) x1[k].resize(nDofx);
        std::vector<double> solu(nDofu);

        for(unsigned i = 0; i < nDofu; i++) {
            unsigned solDof = msh->GetSolutionDof(i, iel, soluType);
            solu[i] = (*sol->_Sol[soluIndex])(solDof);
        }
        for(unsigned i = 0; i < nDofx; i++) {
            unsigned xDof = msh->GetSolutionDof(i, iel, xType);
            for(unsigned k = 0; k < dim; k++) x1[k][i] = (*msh->_topology->_Sol[k])(xDof);
        }

        for(unsigned ig = 0; ig < msh->_finiteElement[ielGeom][soluType]->GetGaussPointNumber(); ig++) {
            msh->_finiteElement[ielGeom][soluType]->Jacobian(x1, ig, weight, phi, phi_x);

            double u = 0.;
            std::vector<double> gradU(dim, 0.), xg(dim, 0.);
            for(unsigned i = 0; i < nDofu; i++) {
                u += phi[i] * solu[i];
                for(unsigned k = 0; k < dim; k++) {
                    gradU[k] += phi_x[i * dim + k] * solu[i];
                    xg[k] += phi[i] * x1[k][i];
                }
            }
            double g2 = 0.;
            for(unsigned k = 0; k < dim; k++) g2 += gradU[k] * gradU[k];
            double g = sqrt(g2);
            if(fabs(u) > maxU) maxU = fabs(u);

            double d = xb - fabs(xg[0]);
            double dy = yb - fabs(xg[1]);
            if(dy < d) d = dy;
            if(d < 0.) d = 0.;

            unsigned b = 0;
            for(unsigned j = 1; j < nf; j++) if(d >= fEdge[j]) b = j;
            if(g > fMax[b]) { fMax[b] = g; fArg[b] = xg; }

            for(unsigned j = 0; j < nm; j++) {
                if(d >= mult[j] * h && g > mMax[j]) { mMax[j] = g; mArg[j] = xg; }
            }
        }
    }

    double maxUAll;
    MPI_Allreduce(&maxU, &maxUAll, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    struct { double val; int rank; } inLoc, outLoc;
    std::vector<double> fMaxAll(nf), mMaxAll(nm);

    for(unsigned b = 0; b < nf; b++) {
        inLoc.val = fMax[b]; inLoc.rank = iproc;
        MPI_Allreduce(&inLoc, &outLoc, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        fMaxAll[b] = outLoc.val;
        MPI_Bcast(fArg[b].data(), dim, MPI_DOUBLE, outLoc.rank, MPI_COMM_WORLD);
    }
    for(unsigned j = 0; j < nm; j++) {
        inLoc.val = mMax[j]; inLoc.rank = iproc;
        MPI_Allreduce(&inLoc, &outLoc, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        mMaxAll[j] = outLoc.val;
        MPI_Bcast(mArg[j].data(), dim, MPI_DOUBLE, outLoc.rank, MPI_COMM_WORLD);
    }

    if(iproc == 0) {
        std::cout.precision(8);
        std::cout << "W1inf over Omega, solution " << solName << ", h = " << h << std::endl;
        std::cout << "  max |u| = " << maxUAll << std::endl;

        std::cout << "  FIXED bands [dist from dOmega)   max |grad u|   argmax" << std::endl;
        for(unsigned b = 0; b < nf; b++) {
            std::cout << "   [" << fEdge[b] << ", ";
            if(b + 1 < nf) std::cout << fEdge[b + 1]; else std::cout << "inf";
            std::cout << ")\t" << fMaxAll[b]
            << "\t(" << fArg[b][0] << ", " << fArg[b][1] << ")" << std::endl;
        }

        std::cout << "  H-SCALED exclusions  sup{|grad u| : d >= m h}   argmax   d/h" << std::endl;
        for(unsigned j = 0; j < nm; j++) {
            double dArg = std::min(xb - fabs(mArg[j][0]), yb - fabs(mArg[j][1]));
            std::cout << "   m = " << mult[j] << "\t" << mMaxAll[j]
            << "\t(" << mArg[j][0] << ", " << mArg[j][1] << ")"
            << "\t" << dArg / h << std::endl;
        }

        double gmax = 0.;
        for(unsigned b = 0; b < nf; b++) if(fMaxAll[b] > gmax) gmax = fMaxAll[b];
        std::cout << "  W1inf(Omega) = " << std::max(maxUAll, gmax) << std::endl;
        std::cout << "  layer max times h = " << fMaxAll[0] * h << std::endl;
    }
}
